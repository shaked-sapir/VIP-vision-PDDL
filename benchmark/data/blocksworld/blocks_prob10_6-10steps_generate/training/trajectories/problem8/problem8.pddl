
(define (problem problem8) (:domain blocks)
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
	
	(holding e)
	(on c b)
	(ontable a)
	(ontable b)
	(ontable d)
  )
  (:goal (and
	(clear a)
	(clear c)
	
	(holding e)
	(on a d)
	(on c b)
	(ontable b)
	(ontable d)))
)
