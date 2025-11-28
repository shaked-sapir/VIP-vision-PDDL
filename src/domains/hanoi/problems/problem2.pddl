(define (problem hanoi2)
  (:domain hanoi)
  (:objects
    peg1 - peg
    peg2 - peg
    peg3 - peg
    d1 - disc
    d2 - disc
  )
  (:init
   (smaller-peg peg1 d1) (smaller-peg peg1 d2)
   (smaller-peg peg2 d1) (smaller-peg peg2 d2)
   (smaller-peg peg3 d1) (smaller-peg peg3 d2)
   (smaller-disc d2 d1)
   (clear-peg peg2) (clear-peg peg3) (clear-disc d1)
   (on-disc d1 d2) (on-peg d2 peg1)
  )
  (:goal (and (on-peg d2 peg3) (on-disc d1 d2)))
  )