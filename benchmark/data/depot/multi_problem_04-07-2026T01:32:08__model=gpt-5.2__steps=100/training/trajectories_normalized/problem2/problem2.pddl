(define (problem depot_d2_p3_t1_p03)
  (:domain depot)

  (:objects
    d1 d2 - depot
    t1 - truck
    c1 c2 - crane
    pile1 pile2 - pile
    p1 p2 p3 - package
  )

  (:init
    (at_truck t1 d1)
    (at_crane c1 d1)
    (empty_crane c1)
    (at_crane c2 d2)
    (empty_crane c2)
    (at_pile pile1 d1)
    (at_pile pile2 d2)
    (on_pile p1 pile1)
    (at p1 d1)
    (on p2 p1)
    (at p2 d1)
    (clear p2)
    (on_pile p3 pile2)
    (at p3 d2)
    (clear p3)
  )

  (:goal
    (and
      (at p1 d2)
      (at p2 d2)
      (at p3 d2)
    )
  )
)
