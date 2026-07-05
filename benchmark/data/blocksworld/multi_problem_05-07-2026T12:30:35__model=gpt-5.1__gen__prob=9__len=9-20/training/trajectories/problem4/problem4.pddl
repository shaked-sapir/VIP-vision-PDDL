
(define (problem problem4) (:domain blocks)
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
	(clear e)
	(handempty)
	(on a b)
	(on b d)
	(ontable c)
	(ontable d)
	(ontable e)
  )
  (:goal (and
	(clear b)
	(clear c)
	(clear d)
	(handfull)
	(holding a)
	(on d e)
	(ontable b)
	(ontable c)
	(ontable e)))
)
