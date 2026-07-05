
(define (problem problem9) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear a)
	(clear c)
	(clear e)
	(handempty)
	(on a d)
	(on c b)
	(ontable b)
	(ontable d)
	(ontable e)
  )
  (:goal (and
	(clear b)
	(clear d)
	(clear e)
	(handempty)
	(on a c)
	(on b a)
	(ontable c)
	(ontable d)
	(ontable e)))
)
