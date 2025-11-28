(define (problem hanoi0)
  (:domain hanoi)
  (:objects
    peg1 - peg
    peg2 - peg
    peg3 - peg
    d1 - disc
    d2 - disc
    d3 - disc
  )
  (:init
   (smaller-peg peg1 d1) (smaller-peg peg1 d2) (smaller-peg peg1 d3)
   (smaller-peg peg2 d1) (smaller-peg peg2 d2) (smaller-peg peg2 d3)
   (smaller-peg peg3 d1) (smaller-peg peg3 d2) (smaller-peg peg3 d3)
   (smaller-disc d2 d1) (smaller-disc d3 d1) (smaller-disc d3 d2)
   (clear-peg peg2) (clear-peg peg3) (clear-disc d1)
   (on-peg d3 peg1) (on-disc d2 d3) (on-disc d1 d2)
  )
  (:goal (and (on-peg d3 peg3) (on-disc d2 d3) (on-disc d1 d2)))
  )