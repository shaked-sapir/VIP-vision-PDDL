
(define (problem problem3) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear c)
	(clear e)
	(handempty)
	(on b d)
	(on c b)
	(on e a)
	(ontable a)
	(ontable d)
  )
  (:goal (and
	(clear c)
	(clear e)
	
	(holding a)
	(on c b)
	(on e d)
	(ontable b)
	(ontable d)))
)
