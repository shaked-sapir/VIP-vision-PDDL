
(define (problem problem2) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear c)
	(clear d)
	
	(holding b)
	(on c e)
	(on e a)
	(ontable a)
	(ontable d)
  )
  (:goal (and
	(clear b)
	(clear e)
	
	(holding c)
	(on b d)
	(on e a)
	(ontable a)
	(ontable d)))
)
