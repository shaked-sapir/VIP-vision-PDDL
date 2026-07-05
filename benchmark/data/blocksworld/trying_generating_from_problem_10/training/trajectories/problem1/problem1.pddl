
(define (problem problem1) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear d)
	(clear e)
	
	(holding c)
	(on d b)
	(on e a)
	(ontable a)
	(ontable b)
  )
  (:goal (and
	(clear b)
	(clear d)
	(handempty)
	(on b c)
	(on c e)
	(on e a)
	(ontable a)
	(ontable d)))
)
