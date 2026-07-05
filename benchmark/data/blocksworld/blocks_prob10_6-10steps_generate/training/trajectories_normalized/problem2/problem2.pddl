
(define (problem problem2) (:domain blocks)
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
	(clear e)
	(handempty)
	(on d b)
	(ontable a)
	(ontable b)
	(ontable c)
	(ontable e)
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
