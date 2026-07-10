
(define (problem problem6) (:domain hanoi)
  (:objects
	d1 - disc
	d2 - disc
	d3 - disc
	peg1 - peg
	peg2 - peg
	peg3 - peg
)
  (:init (clear-disc d1)
	(clear-disc d3)
	(clear-peg peg1)
	(on-disc d1 d2)
	(on-peg d2 peg2)
	(on-peg d3 peg3)
	(smaller-disc d2 d1)
	(smaller-disc d3 d1)
	(smaller-disc d3 d2)
	(smaller-peg peg1 d1)
	(smaller-peg peg1 d2)
	(smaller-peg peg1 d3)
	(smaller-peg peg2 d1)
	(smaller-peg peg2 d2)
	(smaller-peg peg2 d3)
	(smaller-peg peg3 d1)
	(smaller-peg peg3 d2)
	(smaller-peg peg3 d3)
  )
  (:goal (and
	(clear-disc d1)
	(clear-disc d2)
	(clear-disc d3)
	(on-peg d1 peg2)
	(on-peg d2 peg1)
	(on-peg d3 peg3)
	(smaller-disc d2 d1)
	(smaller-disc d3 d1)
	(smaller-disc d3 d2)
	(smaller-peg peg1 d1)
	(smaller-peg peg1 d2)
	(smaller-peg peg1 d3)
	(smaller-peg peg2 d1)
	(smaller-peg peg2 d2)
	(smaller-peg peg2 d3)
	(smaller-peg peg3 d1)
	(smaller-peg peg3 d2)
	(smaller-peg peg3 d3))))
