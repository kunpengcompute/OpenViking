<?php
// Public entry points for the fixture's PHP development server.
$path = rawurldecode(parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH));
if (str_contains($path, '..') || str_contains($path, '\\') || str_contains($path, "\0")) {
    http_response_code(403);
    return true;
}
if ($path === '/' || $path === '/index.php') {
    // Keep native IMAP sorting/pagination; only expose a clearer list control.
    ob_start(static function ($html) {
        if (stripos($html, '<!DOCTYPE html') !== false && strpos($html, 'id="messagelist"') !== false) {
            $version = filemtime(__DIR__ . '/roundcube_sort.js');
            $html = str_ireplace('</body>', '<script src="/office-mail-sort.js?v=' . $version . '"></script></body>', $html);
        }
        return $html;
    });
    require $_SERVER['DOCUMENT_ROOT'] . '/index.php';
    return true;
}
if ($path === '/office-mail-sort.js') {
    header('Content-Type: application/javascript; charset=UTF-8');
    readfile(__DIR__ . '/roundcube_sort.js');
    return true;
}
if (preg_match('#^/(skins|plugins|program/(js|resources))/.*\.(css|js|map|png|jpe?g|gif|svg|ico|eot|ttf|woff2?)$#i', $path)) {
    return false;
}
http_response_code(404);
return true;
