(define (problem gripper-r2-n6-p01)
  (:domain gripper)

  (:objects
    rooma roomb - room
    ball1 ball2 ball3 ball4 ball5 ball6 - ball
    left right - gripper
  )

  (:init
    (at-robby roomb)
    (at ball1 roomb)
    (at ball2 rooma)
    (at ball3 roomb)
    (at ball4 roomb)
    (at ball5 roomb)
    (at ball6 roomb)
    (free left)
    (free right)
  )

  (:goal
    (and
      (at ball1 rooma)
      (at ball2 rooma)
      (at ball3 rooma)
      (at ball4 rooma)
      (at ball5 rooma)
      (at ball6 rooma)
    )
  )
)
