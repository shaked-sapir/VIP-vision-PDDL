
(define (problem problem3) (:domain blocks)
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
	(handfull)
	(holding d)
	(on b c)
	(on c e)
	(ontable a)
	(ontable e)
  )
  (:goal (and
	(clear a)
	(clear e)
	(handfull)
	(holding c)
	(on a b)
	(on b d)
	(ontable d)
	(ontable e)))
)
