"""The web UI's routes, one module per area of the page: events (config, days,
events, cameras, metadata, media), incoming, settings, live, chat, security (the security mode),
health (the warning banner). Each module
has a `routes` table; live and chat keep state of their own in the app
(routes.live.LIVE, routes.chat.CONSOLE)."""
