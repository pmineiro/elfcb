.PHONY: ciorig40 clean

clean:
	rm -f $(wildcard damodel*)

MLE: 
	ln -s ../MLE .

ciorig40: MLE
	./do-ci-shootout.py --dirname orig40

cigt10class: MLE
	./do-ci-shootout.py --dirname gt10class

mlegt10class: MLE
	./do-ci-shootout.py --dirname gt10class --challenger mle
