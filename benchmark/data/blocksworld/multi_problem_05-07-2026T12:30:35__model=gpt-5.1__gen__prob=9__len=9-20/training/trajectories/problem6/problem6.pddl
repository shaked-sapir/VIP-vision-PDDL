
(define (problem problem6) (:domain blocks)
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
	(handfull)
	(holding b)
	(on d a)
	(on e c)
	(ontable a)
	(ontable c)
  )
  (:goal (and
	(clear b)
	(clear e)
	(handfull)
	(holding c)
	(on b d)
	(on d a)
	(ontable a)
	(ontable e)))
)
