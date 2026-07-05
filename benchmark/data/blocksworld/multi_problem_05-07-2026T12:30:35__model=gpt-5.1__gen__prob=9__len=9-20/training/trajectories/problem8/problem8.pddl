
(define (problem problem8) (:domain blocks)
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
	(handfull)
	(holding c)
	(on b d)
	(on d a)
	(ontable a)
	(ontable e)
  )
  (:goal (and
	(clear a)
	(clear d)
	(clear e)
	(handfull)
	(holding c)
	(on d b)
	(ontable a)
	(ontable b)
	(ontable e)))
)
