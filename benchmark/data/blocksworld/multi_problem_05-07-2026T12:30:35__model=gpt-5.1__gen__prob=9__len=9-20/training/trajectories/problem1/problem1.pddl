
(define (problem problem1) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear c)
	(clear e)
	(handempty)
	(on b d)
	(on c b)
	(on d a)
	(ontable a)
	(ontable e)
  )
  (:goal (and
	(clear b)
	(clear c)
	(handempty)
	(on b d)
	(on c e)
	(on d a)
	(ontable a)
	(ontable e)))
)
