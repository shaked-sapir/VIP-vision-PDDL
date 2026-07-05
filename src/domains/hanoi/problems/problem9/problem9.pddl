(define (problem hanoi-4disks-3pegs-p10)
  (:domain hanoi)

  (:objects
    d1 d2 d3 d4 - disk
    a b c - peg
  )

  (:init
    ;; mark places
    (is-disk d1)
    (is-disk d2)
    (is-disk d3)
    (is-disk d4)
    (is-peg a)
    (is-peg b)
    (is-peg c)

    ;; size ordering: d1 < d2 < ... < dN
    (smaller d1 d2)
    (smaller d1 d3)
    (smaller d1 d4)
    (smaller d2 d3)
    (smaller d2 d4)
    (smaller d3 d4)

    ;; initial stacks
    (clear a)
    (on d4 b)
    (on d2 d4)
    (on d1 d2)
    (clear d1)
    (on d3 c)
    (clear d3)
  )

  (:goal
    (and
      (on d4 c)
      (on d3 d4)
      (on d2 d3)
      (on d1 d2)
    )
  )
)
