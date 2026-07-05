
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
	(clear b)
	(clear d)
	(handempty)
	(on a c)
	(on b e)
	(ontable c)
	(ontable d)
	(ontable e)
  )
  (:goal (and
	(clear a)
	(clear c)
	(clear d)
	
	(holding b)
	(on d e)
	(ontable a)
	(ontable c)
	(ontable e)))
)
