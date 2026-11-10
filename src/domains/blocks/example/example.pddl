
(define (problem problem0) (:domain blocks)
  (:objects
        a - block
	b - block
	c - block
	d - block
  )
  (:init 
	(clear a)
	(clear b)
	(clear c)
	(clear d)
	(handempty)
	(ontable a)
	(ontable b)
	(ontable c)
	(ontable d)
  )
  (:goal (and
	(clear c)
	(clear d)
	(handfull)
	(holding a)
	(on c b)
	(ontable b)
	(ontable d)))
)
