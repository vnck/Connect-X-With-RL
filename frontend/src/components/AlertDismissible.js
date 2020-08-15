import React from 'react';
import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';

const AlertDismissible = (props) => {
  return (
      <Alert show={props.show} variant="danger">
        <Alert.Heading>The models feel left out!</Alert.Heading>
        <p>
          We see that you're trying to play a human vs human game. Please be inclusive towards our models, they trained very hard for this!!
        </p>
        <hr />
        <div className="d-flex justify-content-end">
          <Button onClick={() => {props.onHide()}} variant="outline-danger">
            I won't do it again.
          </Button>
        </div>
      </Alert>
  );
}

export default AlertDismissible;