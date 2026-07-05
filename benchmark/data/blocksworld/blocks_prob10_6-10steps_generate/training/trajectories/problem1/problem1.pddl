
(define (problem problem1) (:domain blocks)
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
	(clear e)
	
	(holding d)
	(on a c)
	(ontable b)
	(ontable c)
	(ontable e)
  )
  (:goal (and
	(clear a)
	(clear b)
	(clear c)
	(clear e)
	
	(holding d)
	(ontable a)
	(ontable b)
	(ontable c)
	(ontable e)))
)
