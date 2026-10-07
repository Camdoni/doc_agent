if you want to read data from google maps
Node.js - runtime for async events (in JavaScript)

DO NOT DO THIS (must be fast)
while (( e = getEvent() ) != NULL) {
	handleEvent(e) 
}

SINGLE THREADED - avoid RACES
- You do not have to worry about two different event handlers happening at the same time
- When you are in the event handler you are in charge of the whole computer
- If your thread handler crashes, then your whole computer crashes
- Event handlers do not step on each others' toes

Works by
- specifying event handlers
- i.e. "callbacks"
- x(y) where x is supplied by node
- When x happens, do y

 Python has
 - asyncio
 - Twisted

Ruby has
- Event Machine

We can write a web browser in Node

One of the most common node applications is to write a server

``` node
const http = require('http')
const ip = '127.0.0.1'
const port = 3000

const server = http.createServer ((request, response) => {
	response.statusCode = 200
	response.setHeader('Content-Type', 'text/plain')
	response.end('This is just a toy server.\n')
})

server.listen(port, ip, () => {
	console.log(`Server running at http://${ip}:${port}/`)
})
```

Node packages:
- npm (node package manager)
- it doesn't manage all 3.1 million packages, it manages the packages that your project uses
- what can you do with npm
- if you run npm init, ur saying i want to create a new project and it asks you tell me about your project so i can set things up for you
- it's a little bit interrogative
- npm init --yes 
	- this means fill in all the boxes for me do the default
- package.json
	- this describes the package 
	- json is an object
	- it lists name value pairs
	- scripts has a name / command and a value that has a script to run
	- .
	- .
	- dependencies
	- devDependencies
	- optionalDependencies
		- whether to put something in dependencies or optionalDependencies is your call
	- run/ops time
	- build/dev time
- npm install express
	- you can install an older version using @, express@4.23
	- you might want this because a program u are going to use or create has an older feature that was removed
- npm update
	- what it does is, okay ive been doing this project for a while, but i want to use the most up to date version of my packages
	- syncs you with the outside world
	- can give you bugs
- npm run test
	- when you do this it looks in your package.json and looks for the script name test
	- Helps make it convenient to run

Aside on character encoding:
- Unique characters like an i with two dots, emoji, quotes that curve, japanese characters
- Early characters had (HISTORY)
	- 64 chars - uppercase letters, digits, space, comma, etc (6 bits)
		- 8 bit machines became king because 8 was a nice power of 2
	- 128 chars (7 bits)
		- top bit was a parity bit
	- ASCII
		- has 32 control chars
	- ISO/IEC 8859
		- 8859-1 - has 256 chars, Latin-1
		- 8859-2 central and eastern europe
		- 8859-3 southern europe
		- 8859-5 russian
		- 8859-15 came out in the year 1999 Latin1', added enough to do finnish, estonian, euro symbol
		- http specifies in every sort of message, what the character encoding is,
			- and if you dont then Latin-1 is default
		- when you pick one of these you are stuck with it, 256 chars
- solutions to arbitrary text
	- Unicode - assign a code point to each character 
	- Unicode 17.0 (2025)
		- 159,629 graphic chars
		- bloats our data by a factor of 4
		- now we use 32 bit 
- the vast majority of webpages use multibyte encodings
	- UTF-8 U+0000-U+007F - represented as 0xxxxxxxxx
		- same as ASCII
	- UTF-8 U+0080-U+07FF - represented as 110xxxxxxx
									  10xxxxxxxx
	- UTF-8 U+0800-U+FFFF - represented by a 3 by 5 starts off with a 1110 then 10 10 in each row then everything else is a payload
	- UTF+10000-U+10FFFF
	- these all have header byte and continuation byte
- issues with UTF-8
	- invalid bytes 11111000 F8
	- unexpected continuation 20 A 9C A3
	- truncated sequence         EO A1
	- overlong encoding
- homoglpyhs
	- look same, are different
- synoglpyhs
- normalization - accent chars
	- there are sometimes two different ways to say the same characters