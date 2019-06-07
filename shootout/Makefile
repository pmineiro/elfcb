.PHONY: ciorig40 clean

clean:
	rm -f $(wildcard damodel*)

MLE: 
	ln -s ../MLE .

learningshootoutorig40: MLE
	./do-learning-shootout.py --dirname orig40

learningshootoutgt10class: MLE
	./do-learning-shootout.py --dirname gt10class

learningshootoutmlegt10class: MLE
	./do-learning-shootout.py --dirname gt10class --challenger mle

estimationshootout: MLE
	./do-estimation-shootout.py --dirname orig40
