
(define (problem problem0) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear a)
	(clear b)
	(clear c)
	(clear d)
	(handempty)
	(on d e)
	(ontable a)
	(ontable b)
	(ontable c)
	(ontable e)
  )
  (:goal (and
	(clear a)
	(clear b)
	(clear d)
	(handempty)
	(on a c)
	(on d e)
	(ontable b)
	(ontable c)
	(ontable e)))
)
