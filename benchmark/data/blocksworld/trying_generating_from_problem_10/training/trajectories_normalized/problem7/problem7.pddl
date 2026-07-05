
(define (problem problem7) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear a)
	(clear d)
	(clear e)
	
	(holding b)
	(on a c)
	(ontable c)
	(ontable d)
	(ontable e)
  )
  (:goal (and
	(clear a)
	(clear d)
	(clear e)
	
	(holding b)
	(on a c)
	(ontable c)
	(ontable d)
	(ontable e)))
)
