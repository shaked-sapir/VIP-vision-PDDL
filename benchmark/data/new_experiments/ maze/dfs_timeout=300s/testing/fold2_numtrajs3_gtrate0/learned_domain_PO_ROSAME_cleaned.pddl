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
	:precondition (and (move-dir-up ?from ?to) (clear ?to) (at ?p ?from) (oriented-left ?p))
	:effect (and (move-dir-down ?to ?from) (clear ?from) (at ?p ?to) (oriented-up ?p) (not (move-dir-up ?from ?to))  (not (clear ?to))  (not (at ?p ?from))  (not (oriented-left ?p))))

(:action move_down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-down ?from ?to) (move-dir-left ?from ?to) (move-dir-right ?to ?from) (clear ?to) (oriented-right ?p))
	:effect (and ))

(:action move_left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-right ?to ?from) (clear ?to) (at ?p ?from) (oriented-up ?p))
	:effect (and (move-dir-left ?from ?to) (clear ?from) (at ?p ?to) (oriented-left ?p) (not (clear ?to))  (not (at ?p ?from))  (not (oriented-up ?p))))

(:action move_right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?to) (at ?p ?from) (oriented-up ?p))
	:effect (and (clear ?from) (at ?p ?to) (oriented-left ?p) (not (move-dir-left ?to ?from))  (not (move-dir-right ?from ?to))  (not (clear ?to))  (not (at ?p ?from))  (not (oriented-up ?p))))

)