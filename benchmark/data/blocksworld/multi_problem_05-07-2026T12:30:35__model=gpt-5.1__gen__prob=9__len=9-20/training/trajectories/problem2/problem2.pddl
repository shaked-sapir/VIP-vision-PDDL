
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
	(handfull)
	(holding b)
	(on c e)
	(on d a)
	(ontable a)
	(ontable e)
  )
  (:goal (and
	(clear a)
	(clear b)
	(clear d)
	(handempty)
	(on b c)
	(on c e)
	(ontable a)
	(ontable d)
	(ontable e)))
)
