.PHONY: ciorig40 clean

clean:
	rm -f $(wildcard damodel*)

MLE: 
	ln -s ../MLE .

ciorig40: MLE
	./do-ci-shootout.py --dirname orig40
