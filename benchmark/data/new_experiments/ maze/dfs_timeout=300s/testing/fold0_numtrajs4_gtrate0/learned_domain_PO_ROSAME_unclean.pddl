(define (domain maze)
(:requirements :typing :strips)
(:types 	player location - object
)

(:predicates (move-dir-up ?v0 - location ?v1 - location)
	(move-dir-down ?v0 - location ?v1 - location)
	(move-dir-left ?v0 - location ?v1 - location)
	(move-dir-right ?v0 - location ?v1 - location)
	(clear ?v0 - location)
	(at ?v0 - player ?v1 - location)
	(oriented-up ?v0 - player)
	(oriented-down ?v0 - player)
	(oriented-left ?v0 - player)
	(oriented-right ?v0 - player)
	(is-goal ?v0 - location)
)

(:action move_up
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?to ?from) (move-dir-left ?to ?from) (clear ?from) (clear ?to) (oriented-up ?p) (oriented-down ?p) (oriented-left ?p) (oriented-right ?p))
	:effect (and  (not (move-dir-up ?to ?from))  (not (move-dir-left ?to ?from))  (not (clear ?from))  (not (clear ?to))  (not (oriented-up ?p))  (not (oriented-down ?p))  (not (oriented-left ?p))  (not (oriented-right ?p))))

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?to ?from) (move-dir-down ?from ?to) (clear ?from) (clear ?to) (oriented-up ?p))
	:effect (and (at ?p ?from) (oriented-right ?p) (not (clear ?from))  (not (oriented-up ?p))))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?from ?to) (move-dir-right ?to ?from) (clear ?to) (at ?p ?from) (oriented-right ?p))
	:effect (and (clear ?from) (at ?p ?to) (not (clear ?to))  (not (at ?p ?from))))

(:action move_right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?from) (clear ?to) (oriented-right ?p))
	:effect (and (at ?p ?to) (is-goal ?to) (not (clear ?to))))

)