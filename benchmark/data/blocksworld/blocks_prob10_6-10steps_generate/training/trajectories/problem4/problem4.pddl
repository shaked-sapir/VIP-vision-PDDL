
(define (problem problem4) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear b)
	(clear e)
	
	(holding c)
	(on b d)
	(on e a)
	(ontable a)
	(ontable d)
  )
  (:goal (and
	(clear b)
	(clear c)
	(handempty)
	(on b d)
	(on c e)
	(on e a)
	(ontable a)
	(ontable d)))
)
