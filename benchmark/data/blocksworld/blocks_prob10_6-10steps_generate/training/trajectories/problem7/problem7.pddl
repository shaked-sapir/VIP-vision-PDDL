
(define (problem problem7) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
	e - block
  )
  (:init 
	(clear a)
	(clear e)
	(handempty)
	(on a c)
	(on c b)
	(on e d)
	(ontable b)
	(ontable d)
  )
  (:goal (and
	(clear a)
	(clear c)
	(clear d)
	(clear e)
	(handempty)
	(on c b)
	(ontable a)
	(ontable b)
	(ontable d)
	(ontable e)))
)
