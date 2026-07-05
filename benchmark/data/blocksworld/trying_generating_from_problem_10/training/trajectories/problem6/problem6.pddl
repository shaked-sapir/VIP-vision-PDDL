
(define (problem problem6) (:domain blocks)
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
	
	(holding e)
	(on a d)
	(on c b)
	(ontable b)
	(ontable d)
  )
  (:goal (and
	(clear a)
	(clear b)
	(clear e)
	(handempty)
	(on a c)
	(on b d)
	(ontable c)
	(ontable d)
	(ontable e)))
)
