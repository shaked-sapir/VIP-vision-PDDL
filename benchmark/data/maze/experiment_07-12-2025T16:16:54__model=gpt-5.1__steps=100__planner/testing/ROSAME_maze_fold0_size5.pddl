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

(:action move-up
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?from ?to) (move-dir-down ?to ?from) (clear ?from) (clear ?to) (oriented-left ?p))
	:effect (and (oriented-down ?p) (not (move-dir-up ?from ?to))  (not (move-dir-down ?to ?from))  (not (clear ?from))  (not (oriented-left ?p))))

(:action move-down
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-up ?to ?from) (move-dir-down ?from ?to) (clear ?from) (clear ?to) (oriented-left ?p) (is-goal ?to))
	:effect (and (move-dir-up ?from ?to) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-left ?to ?from) (move-dir-right ?from ?to) (move-dir-right ?to ?from) (at ?p ?from) (at ?p ?to) (oriented-up ?p) (oriented-down ?p) (oriented-right ?p) (is-goal ?from) (not (clear ?to))  (not (oriented-left ?p))  (not (is-goal ?to))))

(:action move-left
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and )
	:effect (and (clear ?from) (at ?p ?from) (oriented-right ?p) (is-goal ?to)))

(:action move-right
	:parameters (?p - player ?from - location ?to - location)
	:precondition (and (move-dir-left ?to ?from) (move-dir-right ?from ?to) (clear ?to) (at ?p ?from) (oriented-down ?p))
	:effect (and (move-dir-up ?from ?to) (move-dir-up ?to ?from) (move-dir-down ?from ?to) (move-dir-down ?to ?from) (move-dir-left ?from ?to) (move-dir-right ?to ?from) (clear ?from) (at ?p ?to) (oriented-up ?p) (oriented-left ?p) (oriented-right ?p) (is-goal ?from) (is-goal ?to) (not (oriented-down ?p))))

)