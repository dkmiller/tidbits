FROM golang:1.22.2 as builder

ENV APP_HOME /go/src/

WORKDIR "$APP_HOME"
COPY ./ .

RUN go mod download
RUN go mod verify
RUN go build .

FROM golang:1.22.2

ENV APP_HOME /go/src/
RUN mkdir -p "$APP_HOME"
WORKDIR "$APP_HOME"
COPY --from=builder "$APP_HOME"/go-newlang $APP_HOME

# EXPOSE 8010
CMD ["./go-newlang"]
