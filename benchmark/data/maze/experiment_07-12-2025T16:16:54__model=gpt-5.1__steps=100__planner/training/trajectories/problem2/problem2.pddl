
(define (problem maze) (:domain maze)
  (:objects
        loc_1_1 - location
	loc_1_2 - location
	loc_1_3 - location
	loc_1_4 - location
	loc_1_5 - location
	loc_1_6 - location
	loc_1_7 - location
	loc_1_8 - location
	loc_2_1 - location
	loc_2_2 - location
	loc_2_3 - location
	loc_2_4 - location
	loc_2_5 - location
	loc_2_6 - location
	loc_2_7 - location
	loc_2_8 - location
	loc_3_1 - location
	loc_3_2 - location
	loc_3_3 - location
	loc_3_4 - location
	loc_3_5 - location
	loc_3_6 - location
	loc_3_7 - location
	loc_3_8 - location
	loc_4_1 - location
	loc_4_2 - location
	loc_4_3 - location
	loc_4_4 - location
	loc_4_5 - location
	loc_4_6 - location
	loc_4_7 - location
	loc_4_8 - location
	loc_5_1 - location
	loc_5_2 - location
	loc_5_3 - location
	loc_5_4 - location
	loc_5_5 - location
	loc_5_6 - location
	loc_5_7 - location
	loc_5_8 - location
	loc_6_1 - location
	loc_6_2 - location
	loc_6_3 - location
	loc_6_4 - location
	loc_6_5 - location
	loc_6_6 - location
	loc_6_7 - location
	loc_6_8 - location
	loc_7_1 - location
	loc_7_2 - location
	loc_7_3 - location
	loc_7_4 - location
	loc_7_5 - location
	loc_7_6 - location
	loc_7_7 - location
	loc_7_8 - location
	loc_8_1 - location
	loc_8_2 - location
	loc_8_3 - location
	loc_8_4 - location
	loc_8_5 - location
	loc_8_6 - location
	loc_8_7 - location
	loc_8_8 - location
	player_1 - player
  )
  (:init 
	(at player_1 loc_2_5)
	(clear loc_2_2)
	(clear loc_2_3)
	(clear loc_2_4)
	(clear loc_2_6)
	(clear loc_3_2)
	(clear loc_3_6)
	(clear loc_3_7)
	(clear loc_4_2)
	(clear loc_4_3)
	(clear loc_4_4)
	(clear loc_4_5)
	(clear loc_4_7)
	(clear loc_5_3)
	(clear loc_5_5)
	(clear loc_5_7)
	(clear loc_6_2)
	(clear loc_6_3)
	(clear loc_6_4)
	(clear loc_6_6)
	(clear loc_6_7)
	(clear loc_7_2)
	(clear loc_7_4)
	(clear loc_7_6)
	(is-goal loc_6_4)
	(move-dir-down loc_2_2 loc_3_2)
	(move-dir-down loc_2_6 loc_3_6)
	(move-dir-down loc_3_2 loc_4_2)
	(move-dir-down loc_3_7 loc_4_7)
	(move-dir-down loc_4_3 loc_5_3)
	(move-dir-down loc_4_5 loc_5_5)
	(move-dir-down loc_4_7 loc_5_7)
	(move-dir-down loc_5_3 loc_6_3)
	(move-dir-down loc_5_7 loc_6_7)
	(move-dir-down loc_6_2 loc_7_2)
	(move-dir-down loc_6_4 loc_7_4)
	(move-dir-down loc_6_6 loc_7_6)
	(move-dir-left loc_2_3 loc_2_2)
	(move-dir-left loc_2_4 loc_2_3)
	(move-dir-left loc_2_5 loc_2_4)
	(move-dir-left loc_2_6 loc_2_5)
	(move-dir-left loc_3_7 loc_3_6)
	(move-dir-left loc_4_3 loc_4_2)
	(move-dir-left loc_4_4 loc_4_3)
	(move-dir-left loc_4_5 loc_4_4)
	(move-dir-left loc_6_3 loc_6_2)
	(move-dir-left loc_6_4 loc_6_3)
	(move-dir-left loc_6_7 loc_6_6)
	(move-dir-right loc_2_2 loc_2_3)
	(move-dir-right loc_2_3 loc_2_4)
	(move-dir-right loc_2_4 loc_2_5)
	(move-dir-right loc_2_5 loc_2_6)
	(move-dir-right loc_3_6 loc_3_7)
	(move-dir-right loc_4_2 loc_4_3)
	(move-dir-right loc_4_3 loc_4_4)
	(move-dir-right loc_4_4 loc_4_5)
	(move-dir-right loc_6_2 loc_6_3)
	(move-dir-right loc_6_3 loc_6_4)
	(move-dir-right loc_6_6 loc_6_7)
	(move-dir-up loc_3_2 loc_2_2)
	(move-dir-up loc_3_6 loc_2_6)
	(move-dir-up loc_4_2 loc_3_2)
	(move-dir-up loc_4_7 loc_3_7)
	(move-dir-up loc_5_3 loc_4_3)
	(move-dir-up loc_5_5 loc_4_5)
	(move-dir-up loc_5_7 loc_4_7)
	(move-dir-up loc_6_3 loc_5_3)
	(move-dir-up loc_6_7 loc_5_7)
	(move-dir-up loc_7_2 loc_6_2)
	(move-dir-up loc_7_4 loc_6_4)
	(move-dir-up loc_7_6 loc_6_6)
	(oriented-right player_1)
  )
  (:goal (and (at player_1 loc_6_4)))
)
