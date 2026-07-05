
(define (problem problem9) (:domain blocks)
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
	(clear d)
	(handempty)
	(on b c)
	(on d e)
	(ontable a)
	(ontable c)
	(ontable e)
  )
  (:goal (and
	(clear a)
	(clear c)
	
	(holding b)
	(on a d)
	(on d e)
	(ontable c)
	(ontable e)))
)
