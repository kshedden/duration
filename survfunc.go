package duration

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/kshedden/dstream/dstream"
)

// SurvfuncRight uses the method of Kaplan and Meier to estimate the
// survival distribution based on (possibly) right censored data.  The
// caller must set Data and TimeVar before calling the Fit method.
// StatusVar, WeightVar, and EntryVar are optional fields.
type SurvfuncRight struct {

	// The data used to perform the estimation.
	data dstream.Dstream

	// The name of the variable containing the minimum of the
	// event time and entry time.  The underlying data must have
	// float64 type.
	timeVar string

	// The name of a variable containing the status indicator,
	// which is 1 if the event occured at the time given by
	// TimeVar, and 0 otherwise.  This is optional, and is assumed
	// to be identically equal to 1 if not present.
	statusVar string

	// The name of a variable containing case weights, optional.
	weightVar string

	// The name of a variable containing entry times, optional.
	entryVar string

	// Times at which events occur, sorted.
	times []float64

	// Number of events at each time in Times.
	nEvents []float64

	// Number of people at risk just before each time in Times
	nRisk []float64

	// The estimated survival function evaluated at each time in Times
	survProb []float64

	// The standard errors for the estimates in SurvProb.
	survProbSE []float64

	events map[float64]float64
	total  map[float64]float64
	entry  map[float64]float64

	timepos   int
	statuspos int
	weightpos int
	entrypos  int
}

// NewSurvfuncRight creates a new value for fitting a survival function.
func NewSurvfuncRight(data dstream.Dstream, timevar, statusvar string) *SurvfuncRight {

	return &SurvfuncRight{
		data:      data,
		timeVar:   timevar,
		statusVar: statusvar,
	}
}

// Weight specifies the name of a case weight variable.
func (sf *SurvfuncRight) Weight(weight string) *SurvfuncRight {
	sf.weightVar = weight
	return sf
}

// Entry specifies the name of an entry time variable.
func (sf *SurvfuncRight) Entry(entry string) *SurvfuncRight {
	sf.entryVar = entry
	return sf
}

// Time returns the times at which the survival function changes.
func (sf *SurvfuncRight) Time() []float64 {
	return sf.times
}

// NumRisk returns the number of people at risk at each time point
// where the survival function changes.
func (sf *SurvfuncRight) NumRisk() []float64 {
	return sf.nRisk
}

// SurvProb returns the estimated survival probabilities at the points
// where the survival function changes.
func (sf *SurvfuncRight) SurvProb() []float64 {
	return sf.survProb
}

// SurvProbSE returns the standard errors of the estimated survival
// probabilities at the points where the survival function changes.
func (sf *SurvfuncRight) SurvProbSE() []float64 {
	return sf.survProbSE
}

func (sf *SurvfuncRight) init() {

	sf.events = make(map[float64]float64)
	sf.total = make(map[float64]float64)
	sf.entry = make(map[float64]float64)

	sf.data.Reset()

	sf.timepos = -1
	sf.statuspos = -1
	sf.weightpos = -1
	sf.entrypos = -1

	for k, na := range sf.data.Names() {
		switch na {
		case sf.timeVar:
			sf.timepos = k
		case sf.statusVar:
			sf.statuspos = k
		case sf.weightVar:
			sf.weightpos = k
		case sf.entryVar:
			sf.entrypos = k
		}
	}

	if sf.timepos == -1 {
		panic("Time variable not found")
	}
	if sf.statuspos == -1 {
		panic("Status variable not found")
	}
	if sf.weightVar != "" && sf.weightpos == -1 {
		panic("Status variable not found")
	}
	if sf.entryVar != "" && sf.entrypos == -1 {
		panic("Entry variable not found")
	}
}

func (sf *SurvfuncRight) scanData() {

	for j := 0; sf.data.Next(); j++ {

		time := sf.data.GetPos(sf.timepos).([]float64)

		var status []float64
		if sf.statuspos != -1 {
			status = sf.data.GetPos(sf.statuspos).([]float64)
		}

		var entry []float64
		if sf.entrypos != -1 {
			entry = sf.data.GetPos(sf.entrypos).([]float64)
		}

		var weight []float64
		if sf.weightpos != -1 {
			weight = sf.data.GetPos(sf.weightpos).([]float64)
		}

		for i, t := range time {

			w := float64(1)
			if sf.weightpos != -1 {
				w = weight[i]
			}

			if sf.statuspos == -1 || status[i] == 1 {
				sf.events[t] += w
			}
			sf.total[t] += w

			if sf.entrypos != -1 {
				if entry[i] >= t {
					msg := fmt.Sprintf("Entry time %d in chunk %d is before the event/censoring times",
						i, j)
					os.Stderr.WriteString(msg)
					os.Exit(1)
				}
				sf.entry[entry[i]] += w
			}
		}
	}
}

func rollback(x []float64) {
	var z float64
	for i := len(x) - 1; i >= 0; i-- {
		z += x[i]
		x[i] = z
	}
}

func (sf *SurvfuncRight) eventstats() {

	// Get the sorted times (event or censoring)
	sf.times = make([]float64, len(sf.total))
	var i int
	for t, _ := range sf.total {
		sf.times[i] = t
		i++
	}
	sort.Sort(sort.Float64Slice(sf.times))

	// Get the weighted event count and risk set size at each time
	// point (in same order as Times).
	sf.nEvents = make([]float64, len(sf.times))
	sf.nRisk = make([]float64, len(sf.times))
	for i, t := range sf.times {
		sf.nEvents[i] = sf.events[t]
		sf.nRisk[i] = sf.total[t]
	}
	rollback(sf.nRisk)

	// Adjust for entry times
	if sf.entrypos != -1 {
		entry := make([]float64, len(sf.times))
		for t, w := range sf.entry {
			ii := sort.SearchFloat64s(sf.times, t)
			if t < sf.times[ii] {
				ii--
			}
			if ii >= 0 {
				entry[ii] += w
			}
		}
		rollback(entry)
		for i := 0; i < len(sf.nRisk); i++ {
			sf.nRisk[i] -= entry[i]
		}
	}
}

// compress removes times where no events occured.
func (sf *SurvfuncRight) compress() {

	var ix []int
	for i := 0; i < len(sf.times); i++ {
		if sf.nEvents[i] > 0 {
			ix = append(ix, i)
		}
	}

	if len(ix) < len(sf.times) {
		for i, j := range ix {
			sf.times[i] = sf.times[j]
			sf.nEvents[i] = sf.nEvents[j]
			sf.nRisk[i] = sf.nRisk[j]
		}
		sf.times = sf.times[0:len(ix)]
		sf.nEvents = sf.nEvents[0:len(ix)]
		sf.nRisk = sf.nRisk[0:len(ix)]
	}
}

func (sf *SurvfuncRight) fit() {

	sf.survProb = make([]float64, len(sf.times))
	x := float64(1)
	for i, _ := range sf.times {
		x *= 1 - sf.nEvents[i]/sf.nRisk[i]
		sf.survProb[i] = x
	}

	sf.survProbSE = make([]float64, len(sf.times))
	x = 0
	if sf.weightpos == -1 {
		for i, _ := range sf.times {
			d := sf.nEvents[i]
			n := sf.nRisk[i]
			x += d / (n * (n - d))
			sf.survProbSE[i] = math.Sqrt(x) * sf.survProb[i]
		}
	} else {
		for i, _ := range sf.times {
			d := sf.nEvents[i]
			n := sf.nRisk[i]
			x += d / (n * n)
			sf.survProbSE[i] = math.Sqrt(x)
		}
	}
}

func (sf *SurvfuncRight) Done() *SurvfuncRight {
	sf.init()
	sf.scanData()
	sf.eventstats()
	sf.compress()
	sf.fit()
	return sf
}
