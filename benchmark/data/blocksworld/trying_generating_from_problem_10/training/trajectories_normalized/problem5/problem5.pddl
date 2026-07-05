
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
	(clear b)
	(clear e)
	
	(holding c)
	(on e d)
	(ontable a)
	(ontable b)
	(ontable d)
  )
  (:goal (and
	(clear a)
	(clear e)
	(handempty)
	(on a d)
	(on c b)
	(on e c)
	(ontable b)
	(ontable d)))
)
