(define (problem hanoi3)
  (:domain hanoi)
  (:objects
    peg1 - peg
    peg2 - peg
    d1 - disc
    d2 - disc
  )
  (:init
   (smaller-peg peg1 d1) (smaller-peg peg1 d2)
   (smaller-peg peg2 d1) (smaller-peg peg2 d2)
   (smaller-disc d2 d1)
   (clear-disc d1) (clear-disc d2)
   (on-peg d1 peg1) (on-peg d2 peg2)
  )
  (:goal (and (on-disc d1 d2)))
  )