
(define (problem problem5) (:domain blocks)
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
	(clear c)
	(clear d)
	(clear e)
	(handempty)
	(on c b)
	(on e a)
	(ontable a)
	(ontable b)
	(ontable d)))
)
