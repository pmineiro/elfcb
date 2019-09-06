.PHONY: ciorig40 clean

clean:
	rm -f $(wildcard damodel*)

MLE: 
	ln -s ../MLE .

learningshootoutorig40: MLE
	eval ./do-learning-shootout.py --dirname orig40 $(extraargs)

learningshootoutcidrorig40: MLE
	eval ./do-learning-shootout.py --dirname orig40 --challenger cidr $(extraargs)

learningshootoutmleorig40: MLE
	eval ./do-learning-shootout.py --dirname orig40 --challenger mle $(extraargs)

learningshootoutmledrorig40: MLE
	eval ./do-learning-shootout.py --dirname orig40 --challenger mledr $(extraargs)

learningshootoutmlecvorig40: MLE
	eval ./do-learning-shootout.py --dirname orig40 --challenger mlecv $(extraargs)

learningshootoutonlineorig40: MLE
	eval ./do-learning-shootout.py --dirname orig40 --challenger onlineci $(extraargs)

learningshootoutgt10class: MLE
	eval ./do-learning-shootout.py --dirname gt10class $(extraargs)

learningshootoutonlinegt10class: MLE
	eval ./do-learning-shootout.py --dirname gt10class --challenger onlineci $(extraargs)

learningshootoutmlegt10class: MLE
	eval ./do-learning-shootout.py --dirname gt10class --challenger mle $(extraargs)

estimationshootout: MLE
	eval ./do-estimation-shootout.py --dirname orig40 $(extraargs)

estimationshootoutcv: MLE
	eval ./do-estimation-shootout.py --dirname orig40 --challenger mlecv $(extraargs)

estimationshootoutdr: MLE
	eval ./do-estimation-shootout.py --dirname orig40 --challenger mledr $(extraargs)

estimationshootoutgt10class: MLE
	eval ./do-estimation-shootout.py --dirname gt10class $(extraargs)

estimationshootoutgt10classcv: MLE
	eval ./do-estimation-shootout.py --dirname gt10class --challenger mlecv $(extraargs)

estimationshootoutgt10classdr: MLE
	eval ./do-estimation-shootout.py --dirname gt10class --challenger mledr $(extraargs)
