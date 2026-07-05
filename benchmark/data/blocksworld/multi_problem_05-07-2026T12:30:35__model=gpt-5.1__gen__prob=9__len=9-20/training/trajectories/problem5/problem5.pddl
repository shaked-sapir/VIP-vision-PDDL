
(define (problem problem5) (:domain blocks)
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
	(clear d)
	(handempty)
	(on a b)
	(on d e)
	(ontable b)
	(ontable c)
	(ontable e)
  )
  (:goal (and
	(clear b)
	(clear d)
	(clear e)
	(handempty)
	(on d a)
	(on e c)
	(ontable a)
	(ontable b)
	(ontable c)))
)
