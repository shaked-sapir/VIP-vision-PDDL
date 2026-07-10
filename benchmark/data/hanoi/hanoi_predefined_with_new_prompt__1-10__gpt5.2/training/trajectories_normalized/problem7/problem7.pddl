
(define (problem problem7) (:domain hanoi)
  (:objects
	d1 - disc
	d2 - disc
	d3 - disc
	peg1 - peg
	peg2 - peg
	peg3 - peg
)
  (:init (clear_disc d1)
	(clear_disc d2)
	(clear_peg peg2)
	(on_disc d1 d3)
	(on_peg d2 peg1)
	(on_peg d3 peg3)
	(smaller_disc d2 d1)
	(smaller_disc d3 d1)
	(smaller_disc d3 d2)
	(smaller_peg peg1 d1)
	(smaller_peg peg1 d2)
	(smaller_peg peg1 d3)
	(smaller_peg peg2 d1)
	(smaller_peg peg2 d2)
	(smaller_peg peg2 d3)
	(smaller_peg peg3 d1)
	(smaller_peg peg3 d2)
	(smaller_peg peg3 d3)
  )
  (:goal (and
	(clear_disc d1)
	(clear_disc d3)
	(clear_peg peg3)
	(on_disc d1 d2)
	(on_peg d2 peg1)
	(on_peg d3 peg2)
	(smaller_disc d2 d1)
	(smaller_disc d3 d1)
	(smaller_disc d3 d2)
	(smaller_peg peg1 d1)
	(smaller_peg peg1 d2)
	(smaller_peg peg1 d3)
	(smaller_peg peg2 d1)
	(smaller_peg peg2 d2)
	(smaller_peg peg2 d3)
	(smaller_peg peg3 d1)
	(smaller_peg peg3 d2)
	(smaller_peg peg3 d3))))
