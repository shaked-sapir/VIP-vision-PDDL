
(define (problem problem0) (:domain strips-sliding-tile)
  (:objects
        t1 - default
	t2 - default
	t3 - default
	t4 - default
	t5 - default
	t6 - default
	t7 - default
	t8 - default
	x1 - default
	x2 - default
	x3 - default
	y1 - default
	y2 - default
	y3 - default
  )
  (:init 
	(at t1 x1 y1)
	(at t2 x3 y1)
	(at t3 x1 y2)
	(at t4 x2 y1)
	(at t5 x3 y2)
	(at t6 x1 y3)
	(at t7 x2 y3)
	(at t8 x3 y3)
	(blank x2 y2)
	(dec x2 x1)
	(dec x3 x2)
	(dec y2 y1)
	(dec y3 y2)
	(inc x1 x2)
	(inc x2 x3)
	(inc y1 y2)
	(inc y2 y3)
	(position x1)
	(position x2)
	(position x3)
	(position y1)
	(position y2)
	(position y3)
	(tile t1)
	(tile t2)
	(tile t3)
	(tile t4)
	(tile t5)
	(tile t6)
	(tile t7)
	(tile t8)
  )
  (:goal (and
	(at t1 x2 y1)
	(at t2 x3 y1)
	(at t3 x1 y1)
	(at t4 x2 y3)
	(at t5 x2 y2)
	(at t6 x1 y2)
	(at t7 x1 y3)
	(at t8 x3 y3)
	(blank x3 y2)
	(dec x2 x1)
	(dec x3 x2)
	(dec y2 y1)
	(dec y3 y2)
	(inc x1 x2)
	(inc x2 x3)
	(inc y1 y2)
	(inc y2 y3)
	(position x1)
	(position x2)
	(position x3)
	(position y1)
	(position y2)
	(position y3)
	(tile t1)
	(tile t2)
	(tile t3)
	(tile t4)
	(tile t5)
	(tile t6)
	(tile t7)
	(tile t8)))
)
