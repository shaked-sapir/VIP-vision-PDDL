
(define (problem problem7) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear b)
	(clear c)
	(handempty)
	(on b d)
	(on c e)
	(on d a)
	(ontable a)
	(ontable e)
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
