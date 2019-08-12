.PHONY: ciorig40 clean

clean:
	rm -f $(wildcard damodel*)

MLE: 
	ln -s ../MLE .

learningshootoutorig40: MLE
	./do-learning-shootout.py --dirname orig40

learningshootoutonlineorig40: MLE
	./do-learning-shootout.py --dirname orig40 --online

learningshootoutgt10class: MLE
	./do-learning-shootout.py --dirname gt10class

learningshootoutonlinegt10class: MLE
	./do-learning-shootout.py --dirname gt10class --online

learningshootoutmlegt10class: MLE
	./do-learning-shootout.py --dirname gt10class --challenger mle

estimationshootout: MLE
	./do-estimation-shootout.py --dirname orig40

estimationshootoutcv: MLE
	./do-estimation-shootout.py --dirname orig40 --challenger mlecv

estimationshootoutgt10class: MLE
	./do-estimation-shootout.py --dirname gt10class

estimationshootoutgt10classcv: MLE
	./do-estimation-shootout.py --dirname gt10class --challenger mlecv
