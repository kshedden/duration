package duration

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/floats"

	"github.com/kshedden/dstream/dstream"
)

// Basic check, no strata, weights, or entry times.
func TestPhreg1(t *testing.T) {

	data := `Time,Status,X
1,1,4
1,1,2
2,0,5
3,0,6
3,1,6
4,0,5
`

	b := bytes.NewBuffer([]byte(data))
	da := dstream.FromCSV(b).SetFloatVars([]string{"Time", "Status", "X"}).HasHeader().Done()
	da = dstream.MemCopy(da)

	ph := NewPHReg(da, "Time", "Status").Done()

	da.Reset()
	phr := NewPHReg(da, "Time", "Status").L2Weight([]float64{0}).Done()

	if fmt.Sprintf("%v", ph.enter) != "[[[0 1 2 3 4 5] []]]" {
		t.Fail()
	}
	if fmt.Sprintf("%v", ph.exit) != "[[[0 1 2] [3 4]]]" {
		t.Fail()
	}
	if fmt.Sprintf("%v", ph.event) != "[[[0 1] [4]]]" {
		t.Fail()
	}

	ll := -14.415134793348063
	for _, pq := range []*PHReg{ph, phr} {
		if math.Abs(pq.breslowLogLike([]float64{2})-ll) > 1e-5 {
			t.Fail()
		}
	}

	ll = -8.9840993267811093
	for _, pq := range []*PHReg{ph, phr} {
		if math.Abs(pq.breslowLogLike([]float64{1})-ll) > 1e-5 {
			t.Fail()
		}
	}

	score := make([]float64, 1)
	sc := -5.66698338
	for _, pq := range []*PHReg{ph, phr} {
		pq.breslowScore([]float64{2}, score)
		if math.Abs(score[0]-sc) > 1e-5 {
			t.Fail()
		}
	}

	sc = -5.09729328
	for _, pq := range []*PHReg{ph, phr} {
		pq.breslowScore([]float64{1}, score)
		if math.Abs(score[0]-sc) > 1e-5 {
			t.Fail()
		}
	}

	hv := -0.93879427
	hess := make([]float64, 1)
	for _, pq := range []*PHReg{ph, phr} {
		pq.breslowHess([]float64{1}, hess)
		if math.Abs(hess[0]-hv) > 1e-5 {
			t.Fail()
		}
	}
}

func TestPhreg2(t *testing.T) {

	data := `Entry,Time,Status,X1,X2,Stratum
0,1,1,4,5,1
1,2,1,2,2,1
0,4,0,3,3,1
1,5,1,5,1,1
3,4,1,1,4,1
2,5,0,3,2,2
1,6,1,5,2,2
2,4,1,4,5,2
1,6,1,2,1,2
3,4,0,6,8,2
5,8,1,6,4,2
`

	b := bytes.NewBuffer([]byte(data))
	vn := []string{"Entry", "Time", "Status", "X1", "X2", "Stratum"}
	da := dstream.FromCSV(b).SetFloatVars(vn).HasHeader().Done()
	da = dstream.MemCopy(da)
	da = dstream.Convert(da, "Stratum", "uint64")
	da = dstream.Regroup(da, "Stratum", true)
	da = dstream.DropCols(da, []string{"Stratum"})

	ph := NewPHReg(da, "Time", "Status").Entry("Entry").Done()

	if fmt.Sprintf("%v", ph.enter) != "[[[0 1 2 3] [] [4] []] [[0 1 2 3 4] [5] []]]" {
		t.Fail()
	}
	if fmt.Sprintf("%v", ph.exit) != "[[[0] [1] [2 4] [3]] [[0 2 4] [1 3] [5]]]" {
		t.Fail()
	}
	if fmt.Sprintf("%v", ph.event) != "[[[0] [1] [4] [3]] [[2] [1 3] [5]]]" {
		t.Fail()
	}

	ll := -26.950282147164277
	if math.Abs(ph.breslowLogLike([]float64{1, 2})-ll) > 1e-5 {
		t.Fail()
	}

	ll = -32.44699788270529
	if math.Abs(ph.breslowLogLike([]float64{2, 1})-ll) > 1e-5 {
		t.Fail()
	}

	score := make([]float64, 2)
	sc := []float64{-9.35565184, -8.0251037}
	ph.breslowScore([]float64{1, 2}, score)
	if !floats.EqualApprox(score, sc, 1e-5) {
		t.Fail()
	}

	sc = []float64{-13.5461984, -3.9178062}
	ph.breslowScore([]float64{2, 1}, score)
	if !floats.EqualApprox(score, sc, 1e-5) {
		t.Fail()
	}

	hess := make([]float64, 4)
	ph.breslowHess([]float64{1, 2}, hess)
	hs := []float64{-1.95989147, 1.23657039, 1.23657039, -1.13182375}
	if !floats.EqualApprox(hess, hs, 1e-5) {
		t.Fail()
	}

	ph.breslowHess([]float64{2, 1}, hess)
	hs = []float64{-1.12887225, 1.21185482, 1.21185482, -2.73825289}
	if !floats.EqualApprox(hess, hs, 1e-5) {
		t.Fail()
	}
}

func TestPhreg3(t *testing.T) {

	var time, status, x1, x2 []float64
	var stratum []uint64

	for i := 0; i < 100; i++ {
		x1 = append(x1, float64(i%3))
		x2 = append(x2, float64(i%7)-3)
		stratum = append(stratum, uint64(i%10))
		if i%5 == 0 {
			status = append(status, 0)
		} else {
			status = append(status, 1)
		}
		time = append(time, 10/float64(4+i%3+i%7-3)+0.5*(float64(i%6)-2))
	}

	dat := [][]interface{}{[]interface{}{time}, []interface{}{status},
		[]interface{}{x1}, []interface{}{x2}, []interface{}{stratum}}
	na := []string{"time", "status", "x1", "x2", "stratum"}
	da := dstream.NewFromArrays(dat, na)
	da = dstream.Regroup(da, "stratum", true)
	da = dstream.DropCols(da, []string{"stratum"})

	ph := NewPHReg(da, "time", "status").Done()
	result, err := ph.Fit()
	if err != nil {
		panic(err)
	}

	// Smoke test
	_ = result.Summary()

	par := result.Params()
	if !floats.EqualApprox(par, []float64{0.1096391, 0.61394886}, 1e-5) {
		t.Fail()
	}

	se := result.StdErr()
	if !floats.EqualApprox(se, []float64{0.17171136, 0.09304276}, 1e-5) {
		t.Fail()
	}
}

// Test whether the results are the same whether we scale or do not
// scale the covariates.
func TestPhregScaling(t *testing.T) {

	data := `Entry,Time,Status,X1,X2,Stratum
0,1,1,4,5,1
1,2,1,2,2,1
0,4,0,3,3,1
1,5,1,5,1,1
3,4,1,1,4,1
2,5,0,3,2,2
1,6,1,5,2,2
2,4,1,4,5,2
1,6,1,2,1,2
3,4,0,6,8,2
5,8,1,6,4,2
`

	b := bytes.NewBuffer([]byte(data))
	vn := []string{"Entry", "Time", "Status", "X1", "X2", "Stratum"}
	da := dstream.FromCSV(b).SetFloatVars(vn).HasHeader().Done()
	da = dstream.MemCopy(da)
	da = dstream.Convert(da, "Stratum", "uint64")
	da = dstream.Regroup(da, "Stratum", true)
	da = dstream.DropCols(da, []string{"Stratum"})

	ph1 := NewPHReg(da, "Time", "Status").Entry("Entry").Done()
	ph2 := NewPHReg(da, "Time", "Status").Entry("Entry").Norm().Done()

	r1, err := ph1.Fit()
	if err != nil {
		panic(err)
	}

	r2, err := ph2.Fit()
	if err != nil {
		panic(err)
	}

	if !floats.EqualApprox(r1.Params(), r2.Params(), 1e-5) {
		t.Fail()
	}
	if !floats.EqualApprox(r1.StdErr(), r2.StdErr(), 1e-5) {
		t.Fail()
	}
}

func TestRegularized(t *testing.T) {

	data := `Time,Status,X1,X2
1,1,4,3
1,1,2,2
2,0,5,2
3,0,6,0
3,1,6,5
4,0,5,4
5,0,4,5
5,1,3,6
6,1,3,5
7,1,5,4
`

	b := bytes.NewBuffer([]byte(data))
	da := dstream.FromCSV(b).SetFloatVars([]string{"Time", "Status", "X1", "X2"}).HasHeader().Done()
	da = dstream.MemCopy(da)

	pe := [][]float64{{-0.499512, -0.127350}, {-0.4776306, -0.11111450}, {-0.4555054, -0.0946655}}

	for j, wt := range []float64{0.01, 0.02, 0.03} {

		l1wgts := []float64{wt, wt}
		ph := NewPHReg(da, "Time", "Status").L1Weight(l1wgts).Done()
		rslt, err := ph.Fit()
		if err != nil {
			panic(err)
		}

		if !floats.EqualApprox(rslt.Params(), pe[j], 1e-5) {
			fmt.Printf("%d\n%v\n", j, rslt.Params())
			t.Fail()
		}
		_ = rslt.Summary()
	}
}
