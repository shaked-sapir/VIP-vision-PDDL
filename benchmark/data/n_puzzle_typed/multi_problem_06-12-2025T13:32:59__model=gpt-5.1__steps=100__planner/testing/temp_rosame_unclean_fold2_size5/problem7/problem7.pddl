
;; eight puzzle problems:
;; hard1 and hard2 are the two "hardest" instances of the puzzle,
;; i.e. having longest solutions (31 steps, see a paper by reinefeld,
;; ijcai -95 or -97).

;; this version uses different sets of objects for x and y coordinates.

(define (problem hard1)
  (:domain n_puzzle_typed)
  (:objects p_1_1 p_1_2 p_1_3 p_2_1 p_2_2 p_2_3 p_3_1 p_3_2 p_3_3 - position t_1 t_2 t_3 t_4 t_5 t_6 t_7 t_8 - tile)
  (:init
   (empty p_2_2) (at t_1 p_1_1) (at t_2 p_3_1) (at t_3 p_1_2)
   (at t_4 p_2_1) (at t_5 p_3_2) (at t_6 p_1_3) (at t_7 p_2_3)
   (at t_8 p_3_3)
   (neighbor p_1_1 p_1_2)
    (neighbor p_1_2 p_1_1)
    (neighbor p_1_2 p_1_3)
    (neighbor p_1_3 p_1_2)
    (neighbor p_2_1 p_2_2)
    (neighbor p_2_2 p_2_1)
    (neighbor p_2_2 p_2_3)
    (neighbor p_2_3 p_2_2)
    (neighbor p_3_1 p_3_2)
    (neighbor p_3_2 p_3_1)
    (neighbor p_3_2 p_3_3)
    (neighbor p_3_3 p_3_2)
    (neighbor p_1_1 p_2_1)
    (neighbor p_2_1 p_1_1)
    (neighbor p_1_2 p_2_2)
    (neighbor p_2_2 p_1_2)
    (neighbor p_1_3 p_2_3)
    (neighbor p_2_3 p_1_3)
    (neighbor p_2_1 p_3_1)
    (neighbor p_3_1 p_2_1)
    (neighbor p_2_2 p_3_2)
    (neighbor p_3_2 p_2_2)
    (neighbor p_2_3 p_3_3)
    (neighbor p_3_3 p_2_3)
   )
  (:goal
   (and (at t_8 p_1_1) (at t_7 p_2_1) (at t_6 p_3_1)
	(at t_4 p_2_2) (at t_1 p_3_2)
	(at t_2 p_1_3) (at t_5 p_2_3) (at t_3 p_3_3)))
  )