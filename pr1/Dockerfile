FROM nginx:1.25-alpine
LABEL maintainer="student@mirea.ru"
LABEL version="1.0"
COPY html/ /usr/share/nginx/html/
COPY nginx/default.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
