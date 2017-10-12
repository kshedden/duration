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
	Data dstream.Dstream

	// The name of the variable containing the minimum of the
	// event time and entry time.  The underlying data must have
	// float64 type.
	TimeVar string

	// The name of a variable containing the status indicator,
	// which is 1 if the event occured at the time given by
	// TimeVar, and 0 otherwise.  This is optional, and is assumed
	// to be identically equal to 1 if not present.
	StatusVar string

	// The name of a variable containing case weights, optional.
	WeightVar string

	// The name of a variable containing entry times, optional.
	EntryVar string

	// Times at which events occur, sorted.
	Times []float64

	// Number of events at each time in Times.
	Events []float64

	// Number of people at risk just before each time in Times
	NRisk []float64

	// The estimated survival function evaluated at each time in Times
	SurvProb []float64

	// The standard errors for the estimates in SurvProb.
	SurvProbSE []float64

	events map[float64]float64
	total  map[float64]float64
	entry  map[float64]float64

	timepos   int
	statuspos int
	weightpos int
	entrypos  int
}

func (sf *SurvfuncRight) init() {

	sf.events = make(map[float64]float64)
	sf.total = make(map[float64]float64)
	sf.entry = make(map[float64]float64)

	sf.Data.Reset()

	sf.timepos = -1
	sf.statuspos = -1
	sf.weightpos = -1
	sf.entrypos = -1

	for k, n := range sf.Data.Names() {
		if n == sf.TimeVar {
			sf.timepos = k
		} else if n == sf.StatusVar {
			sf.statuspos = k
		} else if n == sf.WeightVar {
			sf.weightpos = k
		} else if n == sf.EntryVar {
			sf.entrypos = k
		}
	}

	if sf.timepos == -1 {
		panic("Time variable not found")
	}
}

func (sf *SurvfuncRight) scanData() {

	for j := 0; sf.Data.Next(); j++ {

		time := sf.Data.GetPos(sf.timepos).([]float64)

		var status []float64
		if sf.statuspos != -1 {
			status = sf.Data.GetPos(sf.statuspos).([]float64)
		}

		var entry []float64
		if sf.entrypos != -1 {
			entry = sf.Data.GetPos(sf.entrypos).([]float64)
		}

		var weight []float64
		if sf.weightpos != -1 {
			weight = sf.Data.GetPos(sf.weightpos).([]float64)
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
	sf.Times = make([]float64, len(sf.total))
	var i int
	for t, _ := range sf.total {
		sf.Times[i] = t
		i++
	}
	sort.Sort(sort.Float64Slice(sf.Times))

	// Get the weighted event count and risk set size at each time
	// point (in same order as Times).
	sf.Events = make([]float64, len(sf.Times))
	sf.NRisk = make([]float64, len(sf.Times))
	for i, t := range sf.Times {
		sf.Events[i] = sf.events[t]
		sf.NRisk[i] = sf.total[t]
	}
	rollback(sf.NRisk)

	// Adjust for entry times
	if sf.entrypos != -1 {
		entry := make([]float64, len(sf.Times))
		for t, w := range sf.entry {
			ii := sort.SearchFloat64s(sf.Times, t)
			if t < sf.Times[ii] {
				ii--
			}
			if ii >= 0 {
				entry[ii] += w
			}
		}
		rollback(entry)
		for i := 0; i < len(sf.NRisk); i++ {
			sf.NRisk[i] -= entry[i]
		}
	}
}

// compress removes times where no events occured.
func (sf *SurvfuncRight) compress() {

	var ix []int
	for i := 0; i < len(sf.Times); i++ {
		if sf.Events[i] > 0 {
			ix = append(ix, i)
		}
	}

	if len(ix) < len(sf.Times) {
		for i, j := range ix {
			sf.Times[i] = sf.Times[j]
			sf.Events[i] = sf.Events[j]
			sf.NRisk[i] = sf.NRisk[j]
		}
		sf.Times = sf.Times[0:len(ix)]
		sf.Events = sf.Events[0:len(ix)]
		sf.NRisk = sf.NRisk[0:len(ix)]
	}
}

func (sf *SurvfuncRight) fit() {

	sf.SurvProb = make([]float64, len(sf.Times))
	x := float64(1)
	for i, _ := range sf.Times {
		x *= 1 - sf.Events[i]/sf.NRisk[i]
		sf.SurvProb[i] = x
	}

	sf.SurvProbSE = make([]float64, len(sf.Times))
	x = 0
	if sf.weightpos == -1 {
		for i, _ := range sf.Times {
			d := sf.Events[i]
			n := sf.NRisk[i]
			x += d / (n * (n - d))
			sf.SurvProbSE[i] = math.Sqrt(x) * sf.SurvProb[i]
		}
	} else {
		for i, _ := range sf.Times {
			d := sf.Events[i]
			n := sf.NRisk[i]
			x += d / (n * n)
			sf.SurvProbSE[i] = math.Sqrt(x)
		}
	}
}

func (sf *SurvfuncRight) Fit() {

	sf.init()
	sf.scanData()
	sf.eventstats()
	sf.compress()
	sf.fit()
}
