FROM golang:1.22.2 as builder

ENV APP_HOME /go/src/

WORKDIR "$APP_HOME"
COPY . .

RUN go mod download \
  && go mod verify \
  && go build .

FROM golang:1.22.2

COPY --from=builder /go/src/go-newlang /src/

# EXPOSE 8010
ENTRYPOINT ["/src/go-newlang"]
