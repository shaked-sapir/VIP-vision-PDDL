(define (problem hanoi-4disks-3pegs-p04)
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
    (on d4 a)
    (on d1 d4)
    (clear d1)
    (clear b)
    (on d3 c)
    (on d2 d3)
    (clear d2)
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
