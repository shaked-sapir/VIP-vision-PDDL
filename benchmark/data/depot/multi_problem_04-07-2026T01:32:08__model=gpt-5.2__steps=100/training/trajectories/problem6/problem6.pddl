(define (problem depot-d2-p3-t1-p07)
  (:domain depot)

  (:objects
    d1 d2 - depot
    t1 - truck
    c1 c2 - crane
    pile1 pile2 - pile
    p1 p2 p3 - package
  )

  (:init
    (at-truck t1 d2)
    (at-crane c1 d1)
    (empty-crane c1)
    (at-crane c2 d2)
    (empty-crane c2)
    (at-pile pile1 d1)
    (at-pile pile2 d2)
    (on-pile p1 pile1)
    (at p1 d1)
    (on p3 p1)
    (at p3 d1)
    (clear p3)
    (on-pile p2 pile2)
    (at p2 d2)
    (clear p2)
  )

  (:goal
    (and
      (at p1 d2)
      (at p2 d2)
      (at p3 d2)
    )
  )
)
